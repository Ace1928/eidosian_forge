import re, sys, os, tempfile, json
def phcpy_direct(ideal, tasks=0, precision='d'):
    vars = ideal.ring().variable_names()
    eqns = [repr(p) for p in ideal.gens()]
    return phcpy_direct_base(vars, eqns, tasks=tasks, precision=precision)