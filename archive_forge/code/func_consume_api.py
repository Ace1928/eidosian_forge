import abc
def consume_api(self, agent_api):
    """Consume the AgentAPI instance from the AgentExtensionsManager.

        Allows an extension to gain access to resources internal to the
        neutron agent and otherwise unavailable to the extension.  Examples of
        such resources include bridges, ports, and routers.

        :param agent_api: An instance of an agent-specific API.
        """