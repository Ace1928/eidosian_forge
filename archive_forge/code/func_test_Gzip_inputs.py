from ..misc import Gzip
def test_Gzip_inputs():
    input_map = dict(in_file=dict(extensions=None, mandatory=True), mode=dict(usedefault=True))
    inputs = Gzip.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value