from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@staticmethod
def read_cdft(string: str) -> list[list[dict]]:
    """
        Read cdft parameters from string.

        Args:
            string (str): String

        Returns:
            list[list[dict]]: cdft parameters
        """
    pattern_sec = {'full_section': '\\$cdft((:?(:?\\s*[0-9\\.\\-]+\\s+[0-9]+\\s+[0-9]+(:?\\s+[A-Za-z]+)?\\s*\\n)+|(:?\\s*[0-9\\.\\-]+\\s*\\n)|(:?\\s*\\-+\\s*\\n))+)\\$end'}
    pattern_const = {'constraint': '\\s*([\\-\\.0-9]+)\\s*\\n((?:\\s*(?:[\\-\\.0-9]+)\\s+(?:\\d+)\\s+(?:\\d+)(?:\\s+[A-Za-z]+)?\\s*)+)'}
    section = read_pattern(string, pattern_sec)['full_section']
    if len(section) == 0:
        print('No valid cdft inputs found.')
        return []
    cdft = []
    section = section[0][0]
    states = re.split('\\-{2,25}', section)
    for state in states:
        state_list = []
        const_out = list(read_pattern(state, pattern_const).get('constraint'))
        if len(const_out) == 0:
            continue
        for const in const_out:
            const_dict = {'value': float(const[0]), 'coefficients': [], 'first_atoms': [], 'last_atoms': [], 'types': []}
            sub_consts = const[1].strip().split('\n')
            for subconst in sub_consts:
                tokens = subconst.split()
                const_dict['coefficients'].append(float(tokens[0]))
                const_dict['first_atoms'].append(int(tokens[1]))
                const_dict['last_atoms'].append(int(tokens[2]))
                if len(tokens) > 3:
                    const_dict['types'].append(tokens[3])
                else:
                    const_dict['types'].append(None)
            state_list.append(const_dict)
        cdft.append(state_list)
    return cdft