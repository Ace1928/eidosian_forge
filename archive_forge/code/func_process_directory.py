import sys
import os
import struct
import logging
import numpy as np
def process_directory(request, progressIndicator, readPixelData=False):
    """
    Reads dicom files and returns a list of DicomSeries objects, which
    contain information about the data, and can be used to load the
    image or volume data.

    if readPixelData is True, the pixel data of all series is read. By
    default the loading of pixeldata is deferred until it is requested
    using the DicomSeries.get_pixel_array() method. In general, both
    methods should be equally fast.
    """
    if os.path.isdir(request.filename):
        path = request.filename
    elif os.path.isfile(request.filename):
        path = os.path.dirname(request.filename)
    else:
        raise ValueError('Dicom plugin needs a valid filename to examine the directory')
    files = []
    list_files(files, path)
    series = {}
    count = 0
    progressIndicator.start('examining files', 'files', len(files))
    for filename in files:
        count += 1
        progressIndicator.set_progress(count)
        if filename.count('DICOMDIR'):
            continue
        try:
            dcm = SimpleDicomReader(filename)
        except NotADicomFile:
            continue
        except Exception as why:
            progressIndicator.write(str(why))
            continue
        try:
            suid = dcm.SeriesInstanceUID
        except AttributeError:
            continue
        if suid not in series:
            series[suid] = DicomSeries(suid, progressIndicator)
        series[suid]._append(dcm)
    series = list(series.values())
    series.sort(key=lambda x: x.suid)
    for serie in reversed([serie for serie in series]):
        splitSerieIfRequired(serie, series, progressIndicator)
    series_ = []
    for i in range(len(series)):
        try:
            series[i]._finish()
            series_.append(series[i])
        except Exception as err:
            progressIndicator.write(str(err))
            pass
    progressIndicator.finish('Found %i correct series.' % len(series_))
    return series_