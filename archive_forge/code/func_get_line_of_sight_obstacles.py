from typing import (
import pygame  # Import pygame module for game development
from random import (
import heapq  # Import heapq module for priority queue implementation
import logging  # Import logging module for logging functionality
import math  # Import math module for mathematical operations
def get_line_of_sight_obstacles(self, position: Tuple[int, int], sight_range: int=5) -> Set[Tuple[int, int]]:
    """
        Dynamically calculate obstacles within the line of sight of the agent.

        This function determines the obstacles that are visible to the agent based on its current position and sight range.
        It checks the surrounding positions within the sight range and adds any obstacles found to the set of visible obstacles.

        Args:
            position (Tuple[int, int]): The current position of the agent as a tuple.
            sight_range (int): The range of sight for the agent. Default is 5.

        Returns:
            Set[Tuple[int, int]]: A set of obstacle positions within the line of sight of the agent.
        """
    visible_obstacles: Set[Tuple[int, int]] = set()
    for x in range(max(0, position[0] - sight_range), min(self.grid_width, position[0] + sight_range + 1)):
        for y in range(max(0, position[1] - sight_range), min(self.grid_height, position[1] + sight_range + 1)):
            current_position: Tuple[int, int] = (x, y)
            if current_position == Pathfinder.astar_search.goal_position or current_position in self.body_positions:
                visible_obstacles.add(current_position)
    return visible_obstacles