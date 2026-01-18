import collections
import collections.abc
import operator
import warnings
@classmethod
def from_api_repr(cls, resource):
    """Factory: create a policy from a JSON resource.

        Args:
            resource (dict): policy resource returned by ``getIamPolicy`` API.

        Returns:
            :class:`Policy`: the parsed policy
        """
    version = resource.get('version')
    etag = resource.get('etag')
    policy = cls(etag, version)
    policy.bindings = resource.get('bindings', [])
    for binding in policy.bindings:
        binding['members'] = set(binding.get('members', ()))
    return policy