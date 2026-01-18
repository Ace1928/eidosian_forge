import csv
import io
import logging
import os
import freewilson as fw
from rdkit import Chem, rdBase
def test_chembl():
    logging.getLogger().setLevel(logging.INFO)
    smilesfile = os.path.join(PATH, 'CHEMBL2321810.smi')
    scaffoldfile = os.path.join(PATH, 'CHEMBL2321810_scaffold.mol')
    csvfile = os.path.join(PATH, 'CHEMBL2321810_act.csv')
    assert os.path.exists(smilesfile)
    mols = []
    for line in open(smilesfile):
        smiles, name = line.strip().split()
        m = Chem.MolFromSmiles(smiles)
        m.SetProp('_Name', name)
        mols.append(m)
    scaffold = Chem.MolFromMolBlock(open(scaffoldfile).read())
    data = {k: float(v) for k, v in list(csv.reader(open(csvfile)))[1:]}
    scores = [data[m.GetProp('_Name')] for m in mols]
    assert mols and len(mols) == len(scores)
    with rdBase.BlockLogs():
        free = fw.FWDecompose(scaffold, mols, scores)
    assert free.r2 > 0.8
    preds = list(fw.FWBuild(free))
    assert len(preds)
    preds2 = list(fw.FWBuild(free, pred_filter=lambda x: x > 8))
    assert len(preds2)
    assert len([p for p in preds if p.prediction > 8]) == len(list(preds2))
    s = io.StringIO()
    fw.predictions_to_csv(s, free, preds2)
    assert s.getvalue()
    s2 = io.StringIO(s.getvalue())
    for i, row in enumerate(csv.reader(s2)):
        if i == 0:
            assert row == ['smiles', 'prediction', 'Core_smiles', 'R1_smiles', 'R3_smiles', 'R10_smiles']
    assert i > 0