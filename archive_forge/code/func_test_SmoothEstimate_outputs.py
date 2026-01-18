from ..model import SmoothEstimate
def test_SmoothEstimate_outputs():
    output_map = dict(dlh=dict(), resels=dict(), volume=dict())
    outputs = SmoothEstimate.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value