import numpy as np
import matplotlib.pyplot as plt
import ctypes
import sys
import argparse
import random
def get_simulation_parameters():
    print('\nSimulation Parameter Setup:')
    num_layers = int(input('Enter number of layers (e.g., 3): '))
    num_networks_per_layer = int(input('Enter number of networks per layer (e.g., 6): '))
    num_time_steps = int(input('Enter number of time steps for the simulation (e.g., 50): '))
    num_networks_to_view = int(input('Enter how many mini-networks you would like to view for each layer (e.g., 2): '))
    return (num_layers, num_networks_per_layer, num_time_steps, num_networks_to_view)