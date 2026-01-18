import re, sys, os, tempfile, json
def phc_direct(ideal):
    vars = ideal.ring().variable_names()
    eqns = [repr(p) for p in ideal.gens()]
    return phc_direct_base(vars, eqns)