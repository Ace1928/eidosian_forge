from typing import (
import pygame  # Import pygame module for game development
from random import (
import heapq  # Import heapq module for priority queue implementation
import logging  # Import logging module for logging functionality
import math  # Import math module for mathematical operations
def get_line_of_sight_goals(self, position: Tuple[int, int], sight_range: int=5) -> Set[Tuple[int, int]]:
    """
        Dynamically calculate the goal positions within the line of sight of the agent.

        This function determines the goal positions that are visible to the agent based on its current position and sight range.
        It checks the surrounding positions within the sight range and adds any goal positions found to the set of visible goals.

        Args:
            position (Tuple[int, int]): The current position of the agent as a tuple.
            sight_range (int): The range of sight for the agent. Default is 5.

        Returns:
            Set[Tuple[int, int]]: A set of goal positions within the line of sight of the agent.
        """
    visible_goal_positions: Set[Tuple[int, int]] = set()
    for x in range(max(0, position[0] - sight_range), min(self.grid_width, position[0] + sight_range + 1)):
        for y in range(max(0, position[1] - sight_range), min(self.grid_height, position[1] + sight_range + 1)):
            current_position: Tuple[int, int] = (x, y)
            if current_position in self.astar_search.goal_positions:
                visible_goal_positions.add(current_position)
    return visible_goal_positions