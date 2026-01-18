import abc
from keystone import exception
@abc.abstractmethod
def get_config_option(self, domain_id, group, option, sensitive=False):
    """Get the config option for a domain.

        :param domain_id: the domain for this option
        :param group: the group name
        :param option: the option name
        :param sensitive: whether the option is sensitive

        :returns: dict containing group, option and value
        :raises keystone.exception.DomainConfigNotFound: the option doesn't
                                                         exist.

        """
    raise exception.NotImplemented()