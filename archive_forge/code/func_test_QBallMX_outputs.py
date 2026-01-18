from ..odf import QBallMX
def test_QBallMX_outputs():
    output_map = dict(qmat=dict(extensions=None))
    outputs = QBallMX.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value