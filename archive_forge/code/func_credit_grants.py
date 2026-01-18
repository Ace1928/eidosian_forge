from fastapi import APIRouter, HTTPException, status
from utils.rwkv import AbstractRWKV
import global_var
@router.get('/dashboard/billing/credit_grants', tags=['MISC'])
def credit_grants():
    return {'object': 'credit_summary', 'total_granted': 10000, 'total_used': 0, 'total_available': 10000, 'grants': {'object': 'list', 'data': [{'object': 'credit_grant', 'grant_amount': 10000, 'used_amount': 0, 'effective_at': 1672531200, 'expires_at': 33229440000}]}}