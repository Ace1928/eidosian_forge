from ..dti import DTMetric
def test_DTMetric_inputs():
    input_map = dict(args=dict(argstr='%s'), data_header=dict(argstr='-header %s', extensions=None), eigen_data=dict(argstr='-inputfile %s', extensions=None, mandatory=True), environ=dict(nohash=True, usedefault=True), inputdatatype=dict(argstr='-inputdatatype %s', usedefault=True), metric=dict(argstr='-stat %s', mandatory=True), outputdatatype=dict(argstr='-outputdatatype %s', usedefault=True), outputfile=dict(argstr='-outputfile %s', extensions=None, genfile=True))
    inputs = DTMetric.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value