import os
from random import choice
from itertools import combinations
import snappy
from plink import LinkManager
from .twister_core import build_bundle, build_splitting, twister_version
def splitting(self, gluing, handles, name=None, optimize=True, warnings=True, debugging_level=0, return_type='manifold'):
    """ Generate a manifold with Heegaard splitting this surface from mapping class group data using the Twister
		program of Bell, Hall and Schleimer.
		
		Arguments:
		Required:
		 gluing - the gluing used to join the upper and lower compression bodies
		 handles - where to attach 2-handles
		Optional:
		 name - name of the resulting manifold
		 optimize - try to reduce the number of tetrahedra (default True)
		 warnings - print Twister warnings (default True)
		 debugging_level - specifies the amount of debugging information to be shown (default 0)
		 return_type - specifies how to return the manifold, either as a 'manifold' (default), 'triangulation' or 'string'
		
		Gluing is a word of annulus and rectangle names (or
		their inverses).  These are read from left to right and determine a
		sequence of (half) Dehn twists.  When prefixed with an "!" the name
		specifies a drilling.  For example, "a*B*a*B*A*A*!a*!b" will perform 6
		twists and then drill twice.
		
		Handles is again a word of annulus names (or inverses).  For example,
		'a*c*A' means attach three 2-handles, two above and one below.
		
		Examples:
		
		The genus two splitting of the solid torus:
		>>> M = twister.Surface('S_2').splitting(gluing='', handles='a*B*c') """
    if name is None:
        name = gluing + ' ' + handles
    tri, messages = build_splitting(name, self.surface_contents, gluing, handles, optimize, True, warnings, debugging_level)
    if messages != '':
        print(messages)
    if tri is None:
        return None
    return_type = return_type.lower()
    if return_type == 'manifold':
        return snappy.Manifold(tri)
    if return_type == 'triangulation':
        return snappy.Triangulation(tri)
    if return_type == 'string':
        return tri
    raise TypeError("Return type must be 'manifold', 'triangulation' or 'string'.")