import abc
from tensorflow.python.framework import ops
def get_all_plugin_assets(graph=None):
    """Retrieve all PluginAssets stored in the graph collection.

  Args:
    graph: Optionally, the graph to get assets from. If unspecified, the default
      graph is used.

  Returns:
    A list with all PluginAsset instances in the graph.

  Raises:
    ValueError: if we unexpectedly find a collection with the wrong number of
      PluginAssets.

  """
    if graph is None:
        graph = ops.get_default_graph()
    out = []
    for name in graph.get_collection(_PLUGIN_ASSET_PREFIX):
        collection = graph.get_collection(_PLUGIN_ASSET_PREFIX + name)
        if len(collection) != 1:
            raise ValueError('Collection for %s had %d items, expected 1' % (name, len(collection)))
        out.append(collection[0])
    return out