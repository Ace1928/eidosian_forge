from ..svm import SVMTrain
def test_SVMTrain_outputs():
    output_map = dict(alphas=dict(extensions=None), model=dict(extensions=None), out_file=dict(extensions=None))
    outputs = SVMTrain.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value