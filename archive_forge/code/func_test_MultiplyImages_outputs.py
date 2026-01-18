from ..utils import MultiplyImages
def test_MultiplyImages_outputs():
    output_map = dict(output_product_image=dict(extensions=None))
    outputs = MultiplyImages.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value