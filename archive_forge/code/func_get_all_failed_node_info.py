from typing import List, Set, Tuple
from ray.autoscaler._private import constants
def get_all_failed_node_info(self, non_failed_ids: Set[str]) -> List[Tuple[str, str]]:
    """Get the information about all failed nodes. A failed node is any node which
        we began to track that is not pending or alive (i.e. not failed).

        Args:
            non_failed_ids: Nodes are failed unless they are in this set.

        Returns:
            List[Tuple[str, str]]: A list of tuples. Each tuple is the ip
            address and type of a failed node.
        """
    failed_nodes = self.node_mapping.keys() - non_failed_ids
    failed_info = []
    for node_id in filter(lambda node_id: node_id in failed_nodes, self.lru_order):
        failed_info.append(self.node_mapping[node_id])
    return failed_info