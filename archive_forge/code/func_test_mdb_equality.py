import json
import os
import pyomo.common.unittest as unittest
import pyomo.scripting.pyomo_main as main
from pyomo.opt import check_available_solvers
@unittest.skipUnless(pyodbc_available, 'Requires PyODBC')
def test_mdb_equality(self):
    dat_results_file = self.run_pyomo(os.path.join(exdir, 'diet1.py'), os.path.join(exdir, 'diet.dat'), outputpath=os.path.join(currdir, 'dat_results.jsn'))
    with open(dat_results_file) as FILE:
        dat_results = json.load(FILE)
    db_results_file = self.run_pyomo(os.path.join(exdir, 'diet1.py'), os.path.join(exdir, 'diet1.db.dat'), outputpath=os.path.join(currdir, 'db_results.jsn'))
    with open(db_results_file) as FILE:
        db_results = json.load(FILE)
    del dat_results['Solver'][0]['Time']
    del db_results['Solver'][0]['Time']
    self.assertStructuredAlmostEqual(dat_results, db_results)
    os.remove(dat_results_file)
    os.remove(db_results_file)