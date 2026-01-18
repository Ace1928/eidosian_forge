from ..converters import DWISimpleCompare
def test_DWISimpleCompare_outputs():
    output_map = dict()
    outputs = DWISimpleCompare.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value