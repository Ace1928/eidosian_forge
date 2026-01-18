def test_h2o_xas_xes():
    import ase.calculators.demon as demon
    from ase import Atoms
    import numpy as np
    d = 0.9775
    t = np.pi / 180 * 110.51
    atoms = Atoms('H2O', positions=[(d, 0, 0), (d * np.cos(t), d * np.sin(t), 0), (0, 0, 0)])
    basis = {'all': 'aug-cc-pvdz'}
    auxis = {'all': 'GEN-A2*'}
    input_arguments = {'GRID': 'FINE', 'MOMODIFY': [[1, 0], [1, 0.5]], 'CHARGE': 0, 'XRAY': 'XAS'}
    calc = demon.Demon(basis=basis, auxis=auxis, scftype='UKS TOL=1.0E-6 CDF=1.0E-5', guess='TB', xc=['BLYP', 'BASIS'], input_arguments=input_arguments)
    atoms.calc = calc
    print('XAS hch')
    print('energy')
    energy = atoms.get_potential_energy()
    print(energy)
    ref = -1815.44708987
    error = np.sqrt(np.sum((energy - ref) ** 2))
    print('diff from reference:')
    print(error)
    tol = 0.0001
    assert error < tol
    results = calc.results
    print('xray, first transition, energy')
    value = results['xray']['E_trans'][0]
    print(value)
    ref = 539.410015646
    error = np.sqrt(np.sum((value - ref) ** 2))
    print('diff from reference:')
    print(error)
    tol = 0.0001
    assert error < tol
    print('xray, first transition, transition dipole moments')
    value = results['xray']['trans_dip'][0]
    print(value)
    ref = np.array([0.0111921906, 0.0161393975, 1.70983631e-07])
    error = np.sqrt(np.sum((value - ref) ** 2))
    print('diff from reference:')
    print(error)
    tol = 0.0001
    assert error < tol
    input_arguments = {'GRID': 'FINE', 'CHARGE': 0, 'XRAY': 'XES ALPHA=1-1'}
    calc = demon.Demon(basis=basis, auxis=auxis, scftype='UKS TOL=1.0E-6 CDF=1.0E-5', guess='TB', xc=['BLYP', 'BASIS'], input_arguments=input_arguments)
    atoms.calc = calc
    print('')
    print('XES')
    print('energy')
    energy = atoms.get_potential_energy()
    print(energy)
    ref = -2079.6635944
    error = np.sqrt(np.sum((energy - ref) ** 2))
    print('diff from reference:')
    print(error)
    tol = 0.0001
    assert error < tol
    results = calc.results
    print('xray, first transition, energy')
    value = results['xray']['E_trans'][0]
    print(value)
    ref = 486.862715888
    error = np.sqrt(np.sum((value - ref) ** 2))
    print('diff from reference:')
    print(error)
    tol = 0.0001
    assert error < tol
    print('xray, first transition, transition dipole moments')
    value = results['xray']['trans_dip'][0]
    print(value)
    ref = np.array([0.00650528073, 0.00937895253, 6.9943348e-09])
    error = np.sqrt(np.sum((value - ref) ** 2))
    print('diff from reference:')
    print(error)
    tol = 0.0001
    assert error < tol
    input_arguments = {'GRID': 'FINE', 'MOMODIFY': [[1, 0], [1, 0.0]], 'CHARGE': 0, 'XRAY': 'XAS'}
    calc = demon.Demon(basis=basis, auxis=auxis, scftype='UKS TOL=1.0E-6 CDF=1.0E-5', guess='TB', xc=['BLYP', 'BASIS'], input_arguments=input_arguments)
    atoms.calc = calc
    print('')
    print('XPS')
    print('energy')
    energy = atoms.get_potential_energy()
    print(energy)
    ref = -1536.9295935
    error = np.sqrt(np.sum((energy - ref) ** 2))
    print('diff from reference:')
    print(error)
    tol = 0.0001
    assert error < tol
    input_arguments = {'GRID': 'FINE', 'MOMODIFY': [[1, 0], [1, 0.0]], 'CHARGE': -1}
    calc = demon.Demon(basis=basis, auxis=auxis, scftype='UKS TOL=1.0E-6 CDF=1.0E-5', guess='TB', xc=['BLYP', 'BASIS'], input_arguments=input_arguments)
    atoms.calc = calc
    print('')
    print('EXC')
    print('energy')
    energy = atoms.get_potential_energy()
    print(energy)
    ref = -1543.18092135
    error = np.sqrt(np.sum((energy - ref) ** 2))
    print('diff from reference:')
    print(error)
    tol = 0.0001
    assert error < tol
    print('tests passed')