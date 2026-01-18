from ..preprocess import FitMSParams
def test_FitMSParams_outputs():
    output_map = dict(pd_image=dict(extensions=None), t1_image=dict(extensions=None), t2star_image=dict(extensions=None))
    outputs = FitMSParams.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value