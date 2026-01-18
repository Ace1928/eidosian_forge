from ..preprocess import MRTrixInfo
def test_MRTrixInfo_outputs():
    output_map = dict()
    outputs = MRTrixInfo.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value