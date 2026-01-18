from ..gtract import gtractInvertRigidTransform
def test_gtractInvertRigidTransform_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inputTransform=dict(argstr='--inputTransform %s', extensions=None), numberOfThreads=dict(argstr='--numberOfThreads %d'), outputTransform=dict(argstr='--outputTransform %s', hash_files=False))
    inputs = gtractInvertRigidTransform.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value