from saharaclient.api import parameters as params
def get_node_processes(self, plugin_name, hadoop_version):
    plugin = self.plugins.get_version_details(plugin_name, hadoop_version)
    return self._get_node_processes(plugin)