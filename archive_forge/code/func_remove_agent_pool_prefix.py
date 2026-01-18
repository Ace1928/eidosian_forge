from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import properties
def remove_agent_pool_prefix(agent_pool_string):
    """Removes prefix from transfer agent pool if necessary."""
    prefix_search_result = re.search(_AGENT_POOLS_PREFIX_REGEX, agent_pool_string)
    if prefix_search_result:
        return prefix_search_result.group(2)
    return agent_pool_string