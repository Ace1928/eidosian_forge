from ..dti import FindTheBiggest
def test_FindTheBiggest_outputs():
    output_map = dict(out_file=dict(argstr='%s', extensions=None))
    outputs = FindTheBiggest.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value