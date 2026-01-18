import re
def add_port_mapping(port_bindings, internal_port, external):
    if internal_port in port_bindings:
        port_bindings[internal_port].append(external)
    else:
        port_bindings[internal_port] = [external]