from ..misc import Matlab2CSV
def test_Matlab2CSV_inputs():
    input_map = dict(in_file=dict(extensions=None, mandatory=True), reshape_matrix=dict(usedefault=True))
    inputs = Matlab2CSV.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value