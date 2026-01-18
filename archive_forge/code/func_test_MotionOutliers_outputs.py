from ..utils import MotionOutliers
def test_MotionOutliers_outputs():
    output_map = dict(out_file=dict(extensions=None), out_metric_plot=dict(extensions=None), out_metric_values=dict(extensions=None))
    outputs = MotionOutliers.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value