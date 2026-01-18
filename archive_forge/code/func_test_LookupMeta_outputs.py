from ..dcmstack import LookupMeta
def test_LookupMeta_outputs():
    output_map = dict()
    outputs = LookupMeta.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value