import os.path as op
import nibabel as nb
import numpy as np
from nibabel.volumeutils import native_code
from nibabel.orientations import aff2axcodes
from ... import logging
from ...utils.filemanip import split_filename
from ..base import TraitedSpec, File, isdefined
from ..dipy.base import DipyBaseInterface, HAVE_DIPY as have_dipy
def read_mrtrix_streamlines(in_file, header, as_generator=True):
    offset = header['offset']
    stream_count = header['count']
    fileobj = open(in_file, 'rb')
    fileobj.seek(offset)
    endianness = native_code
    f4dt = np.dtype(endianness + 'f4')
    pt_cols = 3
    bytesize = pt_cols * 4

    def points_per_track(offset):
        track_points = []
        iflogger.info('Identifying the number of points per tract...')
        all_str = fileobj.read()
        num_triplets = int(len(all_str) / bytesize)
        pts = np.ndarray(shape=(num_triplets, pt_cols), dtype='f4', buffer=all_str)
        nonfinite_list = np.where(np.invert(np.isfinite(pts[:, 2])))
        nonfinite_list = list(nonfinite_list[0])[0:-1]
        for idx, value in enumerate(nonfinite_list):
            if idx == 0:
                track_points.append(nonfinite_list[idx])
            else:
                track_points.append(nonfinite_list[idx] - nonfinite_list[idx - 1] - 1)
        return (track_points, nonfinite_list)

    def track_gen(track_points):
        n_streams = 0
        iflogger.info('Reading tracks...')
        while True:
            try:
                n_pts = track_points[n_streams]
            except IndexError:
                break
            pts_str = fileobj.read(n_pts * bytesize)
            nan_str = fileobj.read(bytesize)
            if len(pts_str) < n_pts * bytesize:
                if not n_streams == stream_count:
                    raise nb.trackvis.HeaderError('Expecting %s points, found only %s' % (stream_count, n_streams))
                    iflogger.error('Expecting %s points, found only %s', stream_count, n_streams)
                break
            pts = np.ndarray(shape=(n_pts, pt_cols), dtype=f4dt, buffer=pts_str)
            nan_pt = np.ndarray(shape=(1, pt_cols), dtype=f4dt, buffer=nan_str)
            if np.isfinite(nan_pt[0][0]):
                raise ValueError
                break
            xyz = pts[:, :3]
            yield xyz
            n_streams += 1
            if n_streams == stream_count:
                iflogger.info('100%% : %i tracks read', n_streams)
                raise StopIteration
            try:
                if n_streams % int(stream_count / 100) == 0:
                    percent = int(float(n_streams) / float(stream_count) * 100)
                    iflogger.info('%i%% : %i tracks read', percent, n_streams)
            except ZeroDivisionError:
                iflogger.info('%i stream read out of %i', n_streams, stream_count)
    track_points, nonfinite_list = points_per_track(offset)
    fileobj.seek(offset)
    streamlines = track_gen(track_points)
    if not as_generator:
        streamlines = list(streamlines)
    return streamlines