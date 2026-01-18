from typing import (
import pygame  # Import pygame module for game development
from random import (
import heapq  # Import heapq module for priority queue implementation
import logging  # Import logging module for logging functionality
import math  # Import math module for mathematical operations
def calculate_body_position_proximity_penalty(self, position: Tuple[int, int], body_positions: Set[Tuple[int, int]], space_around_agent: int=2) -> float:
    """
        Calculate a penalty for being too close to the snake's own body.

        This function iterates through each position occupied by the snake's body within the line of sight
        and calculates a penalty if the given position is within a specified distance from any part of the body.
        The penalty is set to infinity to represent an impassable barrier.

        Args:
            position (Tuple[int, int]): The current position as a tuple.
            body_positions (Set[Tuple[int, int]]): The positions occupied by the snake's body.
            space_around_agent (int): The desired space to maintain around the snake's body. Default is 2.

        Returns:
            float: The calculated penalty for being too close to the snake's body.
        """
    penalty: float = 0.0
    visible_body_positions: Set[Tuple[int, int]] = self.get_line_of_sight_body_positions(position)
    for body_position in visible_body_positions:
        if self.calculate_euclidean_distance(position, body_position) < space_around_agent:
            penalty += float('inf')
    return penalty