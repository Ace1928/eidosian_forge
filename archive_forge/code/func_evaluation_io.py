from __future__ import absolute_import, division, print_function
import numpy as np
from . import chords, beats, notes, onsets, tempo
from .beats import BeatEvaluation, BeatMeanEvaluation
from .chords import ChordEvaluation, ChordMeanEvaluation, ChordSumEvaluation
from .key import KeyEvaluation, KeyMeanEvaluation
from .notes import NoteEvaluation, NoteMeanEvaluation, NoteSumEvaluation
from .onsets import OnsetEvaluation, OnsetMeanEvaluation, OnsetSumEvaluation
from .tempo import TempoEvaluation, TempoMeanEvaluation
def evaluation_io(parser, ann_suffix, det_suffix, ann_dir=None, det_dir=None):
    """
    Add evaluation input/output and formatting related arguments to an existing
    parser object.

    Parameters
    ----------
    parser : argparse parser instance
        Existing argparse parser object.
    ann_suffix : str
        Suffix of the annotation files.
    det_suffix : str
        Suffix of the detection files.
    ann_dir : str, optional
        Use only annotations from this folder (and sub-folders).
    det_dir : str, optional
        Use only detections from this folder (and sub-folders).

    Returns
    -------
    io_group : argparse argument group
        Evaluation input / output argument group.
    formatter_group : argparse argument group
        Evaluation formatter argument group.

    """
    import sys
    import argparse
    parser.add_argument('files', nargs='*', help='files (or folders) to be evaluated')
    parser.add_argument('-o', dest='outfile', type=argparse.FileType('w'), default=sys.stdout, help='output file [default: STDOUT]')
    g = parser.add_argument_group('file/folder/suffix arguments')
    g.add_argument('-a', dest='ann_suffix', action='store', default=ann_suffix, help='suffix of the annotation files [default: %(default)s]')
    g.add_argument('--ann_dir', action='store', default=ann_dir, help='search only this directory (recursively) for annotation files [default: %(default)s]')
    g.add_argument('-d', dest='det_suffix', action='store', default=det_suffix, help='suffix of the detection files [default: %(default)s]')
    g.add_argument('--det_dir', action='store', default=det_dir, help='search only this directory (recursively) for detection files [default: %(default)s]')
    g.add_argument('-i', '--ignore_non_existing', action='store_true', help='ignore non-existing detections [default: raise a warning and assume empty detections]')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='increase verbosity level')
    parser.add_argument('-q', '--quiet', action='store_true', help='suppress any warnings')
    parser.set_defaults(output_formatter=tostring)
    f = parser.add_argument_group('formatting arguments')
    formats = f.add_mutually_exclusive_group()
    formats.add_argument('--tex', dest='output_formatter', action='store_const', const=totex, help='format output to be used in .tex files')
    formats.add_argument('--csv', dest='output_formatter', action='store_const', const=tocsv, help='format output to be used in .csv files')
    return (g, f)