import sys
import os.path as op
import pyxnat
from pyxnat.tests import skip_if_no_network
@skip_if_no_network
def test_001_sessionmirror():
    from sessionmirror import create_parser, main, subj_compare
    e = 'BBRCDEV_E03099'
    parser = create_parser()
    args = ['--h1', cfg, '--h2', cfg, '-e', e, '-p', dest_project]
    args = parser.parse_args(args)
    main(args)
    cols = ['label', 'subject_label']
    data = central.array.experiments(experiment_id=e, columns=cols).data[0]
    s1 = central.select.project(src_project).subject(data['subject_label'])
    s2 = central.select.project(dest_project).subject(data['subject_label'])
    assert subj_compare(s1, s2) == 0