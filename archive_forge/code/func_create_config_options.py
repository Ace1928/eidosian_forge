import abc
from keystone import exception
@abc.abstractmethod
def create_config_options(self, domain_id, option_list):
    """Create config options for a domain.

        Any existing config options will first be deleted.

        :param domain_id: the domain for this option
        :param option_list: a list of dicts, each one specifying an option

        Option schema::

            type: dict
            properties:
                group:
                    type: string
                option:
                    type: string
                value:
                    type: depends on the option
                sensitive:
                    type: boolean
            required: [group, option, value, sensitive]
            additionalProperties: false

        """
    raise exception.NotImplemented()