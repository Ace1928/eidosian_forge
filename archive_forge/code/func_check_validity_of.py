from . import exceptions
from . import misc
from . import normalizers
def check_validity_of(self, *components):
    """Check the validity of the components provided.

        This can be specified repeatedly.

        .. versionadded:: 1.1

        :param components:
            Names of components from :attr:`Validator.COMPONENT_NAMES`.
        :returns:
            The validator instance.
        :rtype:
            Validator
        """
    components = [c.lower() for c in components]
    for component in components:
        if component not in self.COMPONENT_NAMES:
            raise ValueError('"{}" is not a valid component'.format(component))
    self.validated_components.update({component: True for component in components})
    return self