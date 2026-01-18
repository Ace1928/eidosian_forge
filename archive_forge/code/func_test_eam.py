import numpy as np
from ase.calculators.eam import EAM
from ase.build import bulk
def test_eam(testdir):
    from scipy.interpolate import InterpolatedUnivariateSpline as spline
    cutoff = 6.28721
    n = 21
    rs = np.arange(0, n) * (cutoff / n)
    rhos = np.arange(0, 2, 2.0 / n)
    m_density = np.array([0.278589606, 0.202694937, 0.145334053, 0.106069912, 0.0842517168, 0.0765140344, 0.0776263116, 0.0823214224, 0.0853322309, 0.0813915861, 0.065909539, 0.0428915711, 0.0227910928, 0.0113713167, 0.00605020311, 0.00365836583, 0.00260587564, 0.00206750708, 0.00148749693, 0.000740019174, 6.21225205e-05])
    m_embedded = np.array([1.04222211e-10, -1.04142633, -1.60359806, -1.89287637, -2.09490167, -2.26456628, -2.40590322, -2.52245359, -2.61385603, -2.67744693, -2.71053295, -2.71110418, -2.69287013, -2.68464527, -2.69204083, -2.68976209, -2.66001244, -2.60122024, -2.51338548, -2.39650817, -2.25058831])
    m_phi = np.array([62.7032242, 34.9638589, 17.9007014, 8.69001383, 4.5154525, 2.83260884, 1.93216616, 1.06795515, 0.337740836, 0.016108789, -0.0620816372, -0.0651314297, -0.0535210341, -0.05209502, -0.0551709524, -0.0489093894, -0.0328051688, -0.0113738785, 0.00233833655, 0.00419132033, 0.000168600692])
    m_densityf = spline(rs, m_density)
    m_embeddedf = spline(rhos, m_embedded)
    m_phif = spline(rs, m_phi)
    a = 4.05
    al = bulk('Al', 'fcc', a=a)
    mishin_approx = EAM(elements=['Al'], embedded_energy=np.array([m_embeddedf]), electron_density=np.array([m_densityf]), phi=np.array([[m_phif]]), cutoff=cutoff, form='alloy', Z=[13], nr=n, nrho=n, dr=cutoff / n, drho=2.0 / n, lattice=['fcc'], mass=[26.982], a=[a])
    al.calc = mishin_approx
    mishin_approx_energy = al.get_potential_energy()
    mishin_approx.write_potential('Al99-test.eam.alloy')
    mishin_check = EAM(potential='Al99-test.eam.alloy')
    al.calc = mishin_check
    mishin_check_energy = al.get_potential_energy()
    print('Cohesive Energy for Al = ', mishin_approx_energy, ' eV')
    error = (mishin_approx_energy - mishin_check_energy) / mishin_approx_energy
    print('read/write check error = ', error)
    assert abs(error) < 0.0001