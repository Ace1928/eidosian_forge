import os
import mido
def print_ports(heading, port_names):
    print(heading)
    for name in port_names:
        print(f"    '{name}'")
    print()