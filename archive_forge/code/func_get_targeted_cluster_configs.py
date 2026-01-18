from saharaclient.api import parameters as params
def get_targeted_cluster_configs(self, plugin_name, hadoop_version):
    plugin = self.plugins.get_version_details(plugin_name, hadoop_version)
    parameters = dict()
    for service in plugin.node_processes.keys():
        parameters[service] = self._extract_parameters(plugin.configs, 'cluster', service)
    return parameters