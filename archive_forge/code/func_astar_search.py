from typing import (
import pygame  # Import pygame module for game development
from random import (
import heapq  # Import heapq module for priority queue implementation
import logging  # Import logging module for logging functionality
import math  # Import math module for mathematical operations
def astar_search(self, start_position: Tuple[int, int], goal_position: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
        Implement the A* search algorithm to find the optimal path from start to goal.

        This function utilizes the A* search algorithm to find the optimal path from the start position to the goal position.
        It maintains an open set of positions to explore, and a closed set of visited positions.
        The algorithm selects the position with the lowest total cost (g_cost + h_cost) from the open set,
        explores its neighbors, and updates the costs and paths accordingly.
        It continues until the goal position is reached or there are no more positions to explore.
        The path is then extended to create a full cycle, zigzagging from the goal back to the start position.

        Args:
            start_position (Tuple[int, int]): The starting position of the search.
            goal_position (Tuple[int, int]): The target position to reach.

        Returns:
            List[Tuple[int, int]]: The optimal path from start to goal and back to start as a list of positions,
                or an empty list if no path is found.
        """
    open_set: List[Tuple[float, float, Tuple[int, int]]] = [(0, 0, start_position)]
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start_position: None}
    g_cost: Dict[Tuple[int, int], float] = {start_position: 0}
    while open_set:
        _, current_g_cost, current_position = heapq.heappop(open_set)
        if current_position == goal_position:
            path_to_goal: List[Tuple[int, int]] = self.reconstruct_path(came_from, start_position, goal_position)
            path_to_start: List[Tuple[int, int]] = self.reconstruct_path(came_from, goal_position, start_position)
            path_to_start.reverse()
            full_path: List[Tuple[int, int]] = path_to_goal + path_to_start
            return full_path
        for neighbor_position in self.get_neighbors(current_position):
            tentative_g_cost: float = current_g_cost + self.calculate_euclidean_distance(current_position, neighbor_position)
            if neighbor_position not in g_cost or tentative_g_cost < g_cost[neighbor_position]:
                g_cost[neighbor_position] = tentative_g_cost
                h_cost: float = self.heuristic(neighbor_position, goal_position)
                total_cost: float = tentative_g_cost + h_cost
                heapq.heappush(open_set, (total_cost, tentative_g_cost, neighbor_position))
                came_from[neighbor_position] = current_position
    return []