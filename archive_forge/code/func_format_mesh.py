from collections.abc import Mapping, Sequence
from subprocess import check_call, DEVNULL
from os import unlink
from pathlib import Path
import numpy as np
from ase.io.utils import PlottingVariables
from ase.constraints import FixAtoms
from ase import Atoms
def format_mesh(self):
    """Returns a formatted data output for POVRAY files

        Example:
        material = '''
          material { // This material looks like pink jelly
            texture {
              pigment { rgbt <0.8, 0.25, 0.25, 0.5> }
              finish{ diffuse 0.85 ambient 0.99 brilliance 3 specular 0.5 roughness 0.001
                reflection { 0.05, 0.98 fresnel on exponent 1.5 }
                conserve_energy
              }
            }
            interior { ior 1.3 }
          }
          photons {
              target
              refraction on
              reflection on
              collect on
          }'''
        """
    if self.material in POVRAY.material_styles_dict:
        material = f'material {{\n        texture {{\n          pigment {{ {pc(self.color)} }}\n          finish {{ {self.material} }}\n        }}\n      }}'
    else:
        material = self.material
    vertex_vectors = self.wrapped_triples_section(triple_list=self.verts, triple_format='<{:f}, {:f}, {:f}>'.format, triples_per_line=4)
    face_indices = self.wrapped_triples_section(triple_list=self.faces, triple_format='<{:n}, {:n}, {:n}>'.format, triples_per_line=5)
    cell = self.cell
    cell_or = self.cell_origin
    mesh2 = f'\n\nmesh2 {{\n    vertex_vectors {{  {len(self.verts):n},\n    {vertex_vectors}\n    }}\n    face_indices {{ {len(self.faces):n},\n    {face_indices}\n    }}\n{(material if material != '' else '// no material')}\n  matrix < {cell[0][0]:f}, {cell[0][1]:f}, {cell[0][2]:f},\n           {cell[1][0]:f}, {cell[1][1]:f}, {cell[1][2]:f},\n           {cell[2][0]:f}, {cell[2][1]:f}, {cell[2][2]:f},\n           {cell_or[0]:f}, {cell_or[1]:f}, {cell_or[2]:f}>\n    }}\n    '
    return mesh2