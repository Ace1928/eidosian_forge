from ..utils import EulerNumber
def test_EulerNumber_outputs():
    output_map = dict(defects=dict(), euler=dict())
    outputs = EulerNumber.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value