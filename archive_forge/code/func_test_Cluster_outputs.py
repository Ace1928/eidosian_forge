from ..model import Cluster
def test_Cluster_outputs():
    output_map = dict(index_file=dict(extensions=None), localmax_txt_file=dict(extensions=None), localmax_vol_file=dict(extensions=None), max_file=dict(extensions=None), mean_file=dict(extensions=None), pval_file=dict(extensions=None), size_file=dict(extensions=None), threshold_file=dict(extensions=None))
    outputs = Cluster.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value