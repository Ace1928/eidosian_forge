from saharaclient.api import parameters as params
def get_cluster_general_configs(self, plugin_name, hadoop_version):
    plugin = self.plugins.get_version_details(plugin_name, hadoop_version)
    return self._extract_parameters(plugin.configs, 'cluster', 'general')