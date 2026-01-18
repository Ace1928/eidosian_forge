from collections import defaultdict
import networkx as nx
def swap(self):
    """Swap left and right intervals"""
    temp = self.left
    self.left = self.right
    self.right = temp