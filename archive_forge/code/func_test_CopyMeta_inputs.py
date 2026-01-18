from ..dcmstack import CopyMeta
def test_CopyMeta_inputs():
    input_map = dict(dest_file=dict(extensions=None, mandatory=True), exclude_classes=dict(), include_classes=dict(), src_file=dict(extensions=None, mandatory=True))
    inputs = CopyMeta.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value