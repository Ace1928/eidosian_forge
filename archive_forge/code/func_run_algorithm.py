from Algorithm import Algorithm
from typing import List, Optional, Dict, Any
import logging
import heapq
from Utility import Node
def run_algorithm(self, snake: 'Snake') -> Optional[List[Node]]:
    """
        Executes the A* algorithm to find the shortest path from the snake's current position to the goal state.

        This method initializes the search space, sets up the initial and goal states, and processes each node
        until the goal is reached or the search space is exhausted. It uses a priority queue to manage the frontier nodes
        and another list to keep track of explored nodes, ensuring that no node is processed more than once.

        Args:
            snake (Snake): The snake object containing the current state of the snake in the game.

        Returns:
            Optional[List[Node]]: A list of Node objects representing the path from the start to the goal state.
            If no path is found, it returns None.
        """
    self.frontier = []
    heapq.heapify(self.frontier)
    self.explored_set = []
    self.path = []
    try:
        initialstate: Node
        goalstate: Node
        initialstate, goalstate = self.get_initstate_and_goalstate(snake)
    except Exception as e:
        logging.error(f'Failed to initialize states in A* algorithm: {str(e)}')
        return None
    heapq.heappush(self.frontier, initialstate)
    logging.debug(f'Initial state {initialstate} added to frontier.')
    while self.frontier:
        try:
            lowest_node: Node = heapq.heappop(self.frontier)
        except Exception as e:
            logging.error(f'Failed to pop from frontier: {str(e)}')
            continue
        logging.debug(f'Lowest node {lowest_node} popped from frontier.')
        try:
            if lowest_node == goalstate:
                logging.info('Goal state reached.')
                path_to_goal = self.get_path(lowest_node)
                logging.debug(f'Path to goal: {path_to_goal}')
                return path_to_goal
        except Exception as e:
            logging.error(f'Error checking goal state or retrieving path: {str(e)}')
            return None
        self.explored_set.append(lowest_node)
        logging.debug(f'Node {lowest_node} added to explored set.')
        try:
            neighbors: List[Node] = self.get_neighbors(lowest_node)
        except Exception as e:
            logging.error(f'Error retrieving neighbors for node {lowest_node}: {str(e)}')
            neighbors = []
        for neighbor in neighbors:
            if self.inside_body(snake, neighbor) or self.outside_boundary(neighbor) or neighbor in self.explored_set:
                logging.debug(f'Skipping neighbor {neighbor} due to invalid conditions.')
                continue
            g: int = lowest_node.g + STEP_COST
            best: bool = False
            if neighbor not in self.frontier:
                neighbor.h = self.manhattan_distance(goalstate, neighbor)
                heapq.heappush(self.frontier, neighbor)
                best = True
                logging.debug(f'Neighbor {neighbor} added to frontier with heuristic {neighbor.h}.')
            else:
                for i in range(len(self.frontier)):
                    if self.frontier[i] == neighbor:
                        if g < self.frontier[i].g:
                            best = True
                            break
            if best:
                neighbor.parent = lowest_node
                neighbor.g = g
                neighbor.f = neighbor.g + neighbor.h
                logging.debug(f'Updated neighbor {neighbor} with new g, f values.')
    logging.info('No path found to goal state.')
    return None