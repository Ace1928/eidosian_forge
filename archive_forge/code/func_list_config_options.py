import abc
from keystone import exception
@abc.abstractmethod
def list_config_options(self, domain_id, group=None, option=False, sensitive=False):
    """Get a config options for a domain.

        :param domain_id: the domain for this option
        :param group: optional group option name
        :param option: optional option name. If group is None, then this
                       parameter is ignored
        :param sensitive: whether the option is sensitive

        :returns: list of dicts containing group, option and value

        """
    raise exception.NotImplemented()