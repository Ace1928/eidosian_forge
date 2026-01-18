import abc
from neutron_lib._i18n import _
from neutron_lib import constants
@classmethod
def get_extended_resources(cls, version):
    """Retrieve the extended resource map for the API definition.

        :param version: The API version to retrieve the resource attribute
            map for.
        :returns: The extended resource map for the underlying API definition
            if the version is 2.0. The extended resource map returned includes
            both the API definition's RESOURCE_ATTRIBUTE_MAP and
            SUB_RESOURCE_ATTRIBUTE_MAP where applicable. If the version is
            not 2.0, an empty dict is returned.
        """
    if version == '2.0':
        cls._assert_api_definition('RESOURCE_ATTRIBUTE_MAP')
        cls._assert_api_definition('SUB_RESOURCE_ATTRIBUTE_MAP')
        sub_attrs = cls.api_definition.SUB_RESOURCE_ATTRIBUTE_MAP or {}
        return dict(list(cls.api_definition.RESOURCE_ATTRIBUTE_MAP.items()) + list(sub_attrs.items()))
    else:
        return {}