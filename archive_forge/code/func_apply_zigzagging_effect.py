from typing import (
import pygame  # Import pygame module for game development
from random import (
import heapq  # Import heapq module for priority queue implementation
import logging  # Import logging module for logging functionality
import math  # Import math module for mathematical operations
def apply_zigzagging_effect(self, current_heuristic: float=1.0) -> float:
    """
        Modify the heuristic to account for zigzagging, making the path less predictable.

        This function increases the heuristic value slightly to account for the added complexity of zigzagging,
        which can make the path less predictable and potentially safer from pursuers.

        Args:
            current_heuristic (float): The current heuristic value. Default is 1.0.

        Returns:
            float: The modified heuristic value accounting for zigzagging.
        """
    return current_heuristic * 1.05