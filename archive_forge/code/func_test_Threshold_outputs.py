from ..model import Threshold
def test_Threshold_outputs():
    output_map = dict(activation_forced=dict(), cluster_forming_thr=dict(), n_clusters=dict(), pre_topo_fdr_map=dict(extensions=None), pre_topo_n_clusters=dict(), thresholded_map=dict(extensions=None))
    outputs = Threshold.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value