from ..preprocess import DARTEL
def test_DARTEL_outputs():
    output_map = dict(dartel_flow_fields=dict(), final_template_file=dict(extensions=None), template_files=dict())
    outputs = DARTEL.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value