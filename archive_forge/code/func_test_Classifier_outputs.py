from ..fix import Classifier
def test_Classifier_outputs():
    output_map = dict(artifacts_list_file=dict(extensions=None))
    outputs = Classifier.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value